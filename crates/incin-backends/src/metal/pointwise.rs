//! Elementwise Metal operations: tape-tracked unaries, scalars, clamp, and
//! the three broadcast binaries (#92 Batch A).
//!
//! Every method here is pure host-side `Vec<f32>` math over
//! [`MetalStorage`]'s shared bytes — the same shape as `backend.rs`'s
//! existing `add`/`mul` paths — so the module compiles and its tests run on
//! any host under `--features metal`. Recipes mirror the CPU canonical
//! kernels (`cpu/ops/elementwise/{unary,binary}.rs`) so both backends agree
//! on the point each derivative is evaluated at, not merely on its shape.

use incin_core::backend_authoring::*;
use incin_core::error::Result;
use incin_core::tensor::device::Device;
use incin_core::tensor::dtype::DType;

use super::backend::MetalBackendImpl;
use super::backend::{
    binary_op_metal, scalar_op_metal, storage_from_f32, unary_op_metal, unbroadcast,
};
use super::storage::MetalStorage;

/// Push a single-input `TapeEntry` whose backward maps one cotangent to one
/// gradient. Shared by every unary/scalar recipe below so the
/// `TapeEntry { output_id, input_ids: vec![t.id], backward: ... }`
/// boilerplate is written once.
fn push_unary_tape_entry(
    t_id: incin_core::exec::TensorId,
    out_id: incin_core::exec::TensorId,
    grad_fn: impl Fn(&MetalStorage) -> Result<MetalStorage> + Send + Sync + 'static,
) {
    crate::metal::tape::push(crate::metal::tape::TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out| grad_fn(grad_out).map(|grad| vec![grad])),
    });
}

/// Abramowitz & Stegun 7.1.26 error-function approximation.
///
/// Copied from `cpu/ops/elementwise_kernel/util.rs` (`pub(super)` there, so
/// unreachable from this module). `gelu`/`erf` must use the *same* polynomial
/// as the CPU reference or the two backends disagree on every non-integer
/// input; see the CPU util's doc for the coefficient provenance.
fn erf_approx_f64(value: f64) -> f64 {
    let sign = if value < 0.0 { -1.0 } else { 1.0 };
    let value = value.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * value);
    let polynomial =
        (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t;
    sign * (1.0 - polynomial * (-value * value).exp())
}

/// Softplus with the large-input shortcut CPU's `Mish` kernel uses.
fn softplus_f32(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        (1.0 + value.exp()).ln()
    }
}

/// `sign(x)` as CPU's `UnaryOp::Sign` evaluates it (used by `abs`'s recipe).
fn sign_f32(value: f32) -> f32 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// `step(x)` as CPU's `UnaryOp::Step` evaluates it (used by `relu`'s recipe).
fn step_f32(value: f32) -> f32 {
    if value > 0.0 { 1.0 } else { 0.0 }
}

// ── Dropout counter hash (#84's reproducible draws) ─────────────────────────
//
// Duplicated from `cuda/ops/dropout.rs`: the two backends are separate
// feature gates, so `cuda` is not compiled under `--features metal` and the
// helpers cannot be shared from there. The mix, the seed and the advancing
// flat-index offset are copied value for value so a (seed, index) pair means
// the same draw on either backend.

/// Module-level seed for every dropout draw on this process.
///
/// A fixed constant so tests are deterministic out of the box; the draw
/// offset below advances past it on every call so consecutive dropouts on
/// the same seed still see disjoint index ranges.
static DROPOUT_SEED: core::sync::atomic::AtomicU64 =
    core::sync::atomic::AtomicU64::new(0x9E37_79B9_7F4A_7C15);

/// Monotonic flat-index offset advanced by `numel` on every draw.
static DROPOUT_OFFSET: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// SplitMix64 finalizer: the mix that turns a (seed, index) pair into a
/// well-scattered 64-bit value. Kept as a pure function so tests can call it
/// directly without touching atomics.
#[must_use]
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Map `(seed, index)` to a uniform `f32` in `[0, 1)`.
///
/// Deterministic, pure, and independent of call order: the same pair always
/// yields the same float. The high 24 bits of the mix are shifted into the
/// mantissa of a `1.0` bit pattern, then `1.0` is subtracted — the standard
/// "u32 to unit float" trick that keeps every bit of entropy the hash
/// produced.
#[must_use]
fn hash_uniform(seed: u64, index: u64) -> f32 {
    let mixed = mix64(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(index));
    let bits = (mixed >> 40) as u32; // top 24 bits -> mantissa
    f32::from_bits(0x3F80_0000 | bits) - 1.0
}

/// Reserve `numel` consecutive counter indices and return
/// `(seed, start_index)`.
fn reserve_draw(numel: u64) -> (u64, u64) {
    let seed = DROPOUT_SEED.load(core::sync::atomic::Ordering::Relaxed);
    let start = DROPOUT_OFFSET.fetch_add(numel, core::sync::atomic::Ordering::Relaxed);
    (seed, start)
}

/// Forward `f`, then record `backward(t, out, grad_out) -> grad`.
///
/// Most unaries below need both the captured input and the captured output
/// in their recipe (output-based derivatives evaluate at `out`, input-based
/// ones at `t`, and `swish`/`elu` need both). Writing the push once keeps the
/// two captures from being forgotten at a call site.
macro_rules! unary_taped {
    ($method:ident, $forward:expr, $backward:expr) => {
        pub(crate) fn $method<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let out = unary_op_metal(t, $forward)?;
            let t_capture = t.clone();
            let out_capture = out.clone();
            push_unary_tape_entry(t.id(), out.id(), move |grad_out| {
                ($backward)(&t_capture, &out_capture, grad_out)
            });
            Ok(out)
        }
    };
}

/// Push a tape entry whose backward is `grad_out * f'(x)` for a unary whose
/// derivative is a plain function of the input (the CPU
/// `canonical_unary_with_deriv_op` family).
macro_rules! unary_input_deriv_taped {
    ($method:ident, $forward:expr, $deriv:expr) => {
        unary_taped!($method, $forward, |t, _out, grad_out| {
            let deriv = unary_op_metal(t, $deriv)?;
            binary_op_metal(
                grad_out,
                &deriv,
                concat!(stringify!($method), "_grad"),
                |g, d| g * d,
            )
        });
    };
}

/// Push a tape entry whose backward is `grad_out * f'(out)` for a unary whose
/// derivative is a plain function of the output.
macro_rules! unary_output_deriv_taped {
    ($method:ident, $forward:expr, $deriv:expr) => {
        unary_taped!($method, $forward, |_t, out, grad_out| {
            let deriv = unary_op_metal(out, $deriv)?;
            binary_op_metal(
                grad_out,
                &deriv,
                concat!(stringify!($method), "_grad"),
                |g, d| g * d,
            )
        });
    };
}

/// Training-false unaries (and `trunc`) record a zero gradient so a graph
/// containing one still walks — matching WGPU, and matching CPU's
/// `canonical_step`. The value is all-zero at the cotangent's shape.
macro_rules! unary_zero_grad_taped {
    ($method:ident, $forward:expr) => {
        unary_taped!($method, $forward, |_t, _out, grad_out| {
            unary_op_metal(grad_out, |_| 0.0)
        });
    };
}

// The unary/binary constructors below are macro invocations that expand to
// `fn` items; rustc treats `///` on the invocation itself as an unused doc
// comment even though the expansion is what readers of this file need the
// docs for. Suppress the lint at the impl boundary rather than degrading
// those docs to `//`.
#[allow(unused_doc_comments)]
impl<D: Device> MetalBackendImpl<D> {
    // ── Activations & elementwise basics ───────────────────────────────────

    /// `relu`. Derivative is `step(x)` at the input (CPU `canonical_relu`).
    unary_taped!(relu, |x| x.max(0.0), |t, _out, g| {
        let mask = unary_op_metal(t, step_f32)?;
        binary_op_metal(g, &mask, "relu_grad", |a, b| a * b)
    });

    /// `step`. Flat almost everywhere (`training = false`); records zeros.
    unary_zero_grad_taped!(step, step_f32);

    /// `mish = x * tanh(softplus(x))`. Derivative at the input, CPU's
    /// `MishBackward`: `tanh(sp) + x * sigmoid(x) * (1 - tanh^2)`.
    unary_input_deriv_taped!(mish, |x| x * softplus_f32(x).tanh(), |x| {
        let softplus = softplus_f32(x);
        let tanh = softplus.tanh();
        let sigmoid = 1.0 / (1.0 + (-x).exp());
        tanh + x * sigmoid * (1.0 - tanh * tanh)
    });

    /// `elu`. Derivative at the *output* (CPU `canonical_elu`): `out > 0 → 1`,
    /// else `out + 1` (which equals `exp(x)` for the negative branch).
    unary_output_deriv_taped!(elu, |x| if x > 0.0 { x } else { x.exp() - 1.0 }, |out| {
        if out > 0.0 { 1.0 } else { out + 1.0 }
    });

    /// `gelu` (erf form). Derivative at the input: `cdf(x) + x * pdf(x)`,
    /// evaluated in `f64` with the same `erf_approx_f64` as the forward.
    unary_input_deriv_taped!(
        gelu,
        |x: f32| {
            let x64 = f64::from(x);
            (x64 * 0.5 * (1.0 + erf_approx_f64(x64 / core::f64::consts::SQRT_2))) as f32
        },
        |x: f32| {
            let x64 = f64::from(x);
            let cdf = 0.5 * (1.0 + erf_approx_f64(x64 / core::f64::consts::SQRT_2));
            let pdf = (1.0 / (2.0 * core::f64::consts::PI).sqrt()) * (-x64 * x64 / 2.0).exp();
            (cdf + x64 * pdf) as f32
        }
    );

    /// `abs`. Derivative is `sign(x)` at the input.
    unary_taped!(abs, |x| x.abs(), |t, _out, g| {
        let mask = unary_op_metal(t, sign_f32)?;
        binary_op_metal(g, &mask, "abs_grad", |a, b| a * b)
    });

    /// `exp`. Output-based: `grad * out`.
    unary_output_deriv_taped!(exp, |x| x.exp(), |out| out);

    /// `neg`. Constant derivative `-1`.
    unary_taped!(neg, |x| -x, |_t, _out, g| unary_op_metal(g, |x| -x));

    /// `sqrt`. Output-based: `grad / out * 0.5`.
    unary_taped!(sqrt, |x| x.sqrt(), |_t, out, g| {
        let ratio = binary_op_metal(g, out, "sqrt_grad_ratio", |a, b| a / b)?;
        scalar_op_metal(&ratio, 0.5, |x, s| x * s)
    });

    /// `log`. Input-based: `grad / x`.
    unary_taped!(log, |x| x.ln(), |t, _out, g| {
        binary_op_metal(g, t, "log_grad", |a, b| a / b)
    });

    /// `tanh`. Output-based: `grad * (1 - out^2)`.
    unary_output_deriv_taped!(tanh, |x| x.tanh(), |out| 1.0 - out * out);

    /// `sigmoid`. Output-based: `grad * out * (1 - out)`.
    unary_output_deriv_taped!(sigmoid, |x| 1.0 / (1.0 + (-x).exp()), |out| {
        out * (1.0 - out)
    });

    /// `swish = x * sigmoid(x)`. Derivative `out + sigmoid(x) * (1 - out)`,
    /// needing both captures (CPU's f64 index-walk recipe).
    unary_taped!(swish, |x| x / (1.0 + (-x).exp()), |t, out, g| {
        let sig = unary_op_metal(t, |x| 1.0 / (1.0 + (-x).exp()))?;
        let one_minus_out = unary_op_metal(out, |o| 1.0 - o)?;
        let sig_term = binary_op_metal(&sig, &one_minus_out, "swish_grad_sig", |s, o| s * o)?;
        let deriv = binary_op_metal(out, &sig_term, "swish_grad_deriv", |o, d| o + d)?;
        binary_op_metal(g, &deriv, "swish_grad", |a, b| a * b)
    });

    // ── Training-false / zero-derivative unaries ───────────────────────────

    /// `sign`. Training-false; records zeros.
    unary_zero_grad_taped!(sign, sign_f32);
    /// `floor`. Training-false; records zeros.
    unary_zero_grad_taped!(floor, |x| x.floor());
    /// `ceil`. Training-false; records zeros.
    unary_zero_grad_taped!(ceil, |x| x.ceil());
    /// `round`. Training-false; records zeros.
    unary_zero_grad_taped!(round, |x| x.round());

    // ── Logarithms ─────────────────────────────────────────────────────────

    /// `log2`. Derivative `1 / (x * ln 2)`.
    unary_input_deriv_taped!(log2, |x| x.log2(), |x| 1.0 / (x * core::f32::consts::LN_2));
    /// `log10`. Derivative `1 / (x * ln 10)`.
    unary_input_deriv_taped!(log10, |x| x.log10(), |x| {
        1.0 / (x * core::f32::consts::LN_10)
    });

    // ── Trigonometry ───────────────────────────────────────────────────────

    /// `sin`. Derivative `cos(x)`.
    unary_input_deriv_taped!(sin, |x| x.sin(), |x| x.cos());
    /// `cos`. Derivative `-sin(x)`.
    unary_input_deriv_taped!(cos, |x| x.cos(), |x| -x.sin());
    /// `tan`. Derivative `1 + tan^2(x)` — evaluated at the input as CPU's
    /// `TanBackward` does (recomputing `tan(x)` there).
    unary_input_deriv_taped!(tan, |x| x.tan(), |x| 1.0 + x.tan().powi(2));
    /// `asin`. Derivative `1 / sqrt(1 - x^2)`.
    unary_input_deriv_taped!(asin, |x| x.asin(), |x| 1.0 / (1.0 - x * x).sqrt());
    /// `acos`. Derivative `-1 / sqrt(1 - x^2)`.
    unary_input_deriv_taped!(acos, |x| x.acos(), |x| -1.0 / (1.0 - x * x).sqrt());
    /// `atan`. Derivative `1 / (1 + x^2)`.
    unary_input_deriv_taped!(atan, |x| x.atan(), |x| 1.0 / (1.0 + x * x));
    /// `sinh`. Derivative `cosh(x)`.
    unary_input_deriv_taped!(sinh, |x| x.sinh(), |x| x.cosh());
    /// `cosh`. Derivative `sinh(x)`.
    unary_input_deriv_taped!(cosh, |x| x.cosh(), |x| x.sinh());
    /// `asinh`. Derivative `1 / sqrt(x^2 + 1)`.
    unary_input_deriv_taped!(asinh, |x| x.asinh(), |x| 1.0 / (x * x + 1.0).sqrt());
    /// `acosh`. Derivative `1 / sqrt(x^2 - 1)`.
    unary_input_deriv_taped!(acosh, |x| x.acosh(), |x| 1.0 / (x * x - 1.0).sqrt());
    /// `atanh`. Derivative `1 / (1 - x^2)`.
    unary_input_deriv_taped!(atanh, |x| x.atanh(), |x| 1.0 / (1.0 - x * x));

    // ── Special functions ──────────────────────────────────────────────────

    /// `erf`. Forward uses the shared `erf_approx_f64`; derivative
    /// `(2 / sqrt(pi)) * exp(-x^2)` at the input (CPU `ErfBackward`).
    unary_input_deriv_taped!(
        erf,
        |x: f32| erf_approx_f64(f64::from(x)) as f32,
        |x: f32| (2.0 / core::f32::consts::PI.sqrt()) * (-x * x).exp()
    );

    /// `rsqrt(x) = 1 / sqrt(x)`. Derivative `-0.5 / (x * sqrt(x))`.
    unary_input_deriv_taped!(rsqrt, |x| 1.0 / x.sqrt(), |x| -0.5 / (x * x.sqrt()));

    /// `trunc`. Derivative zero wherever it exists; records that zero (the
    /// row is `training = true` and a training graph that records no node
    /// comes apart — WGPU records the same zero).
    unary_zero_grad_taped!(trunc, |x| x.trunc());

    /// `frac(x) = x - trunc(x)`. Derivative 1 wherever it exists; the
    /// gradient passes straight through (CPU `canonical_frac`).
    unary_taped!(frac, |x| x.fract(), |_t, _out, g: &MetalStorage| Ok(
        g.clone()
    ));

    // ── Scalars ────────────────────────────────────────────────────────────

    /// `sub_scalar(x, c)`. Derivative is the identity.
    pub(crate) fn sub_scalar_float<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        scalar: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = scalar_op_metal(t, scalar, |x, s| x - s)?;
        push_unary_tape_entry(t.id(), out.id(), |g| Ok(g.clone()));
        Ok(out)
    }

    /// `div_scalar(x, c)`. Derivative `1 / c`, applied by the same division
    /// the forward used so the constant's rounding matches.
    pub(crate) fn div_scalar_float<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        scalar: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = scalar_op_metal(t, scalar, |x, s| x / s)?;
        push_unary_tape_entry(t.id(), out.id(), move |g| {
            scalar_op_metal(g, scalar, |x, s| x / s)
        });
        Ok(out)
    }

    /// `powf(x, p)` with the exponent as a scalar attribute. Backward is
    /// `p * x^(p-1)` at the captured input (CPU `canonical_powf`).
    pub(crate) fn powf<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        exponent: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = scalar_op_metal(t, exponent, |x, s| x.powf(s))?;
        let t_capture = t.clone();
        push_unary_tape_entry(t.id(), out.id(), move |g| {
            let derivative = scalar_op_metal(&t_capture, exponent - 1.0, |x, s| x.powf(s))?;
            let scaled = scalar_op_metal(&derivative, exponent, |x, s| x * s)?;
            binary_op_metal(g, &scaled, "powf_grad", |a, b| a * b)
        });
        Ok(out)
    }

    /// `clamp(x, min, max)`. Cotangent passes through the interior and stops
    /// outside both bounds; on a boundary CPU's condition (`value < min ||
    /// value > max`) is false, so the gradient passes — matching the code.
    pub(crate) fn clamp<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        min: f64,
        max: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let min_f = min as f32;
        let max_f = max as f32;
        let out = unary_op_metal(t, move |x| x.clamp(min_f, max_f))?;
        let t_capture = t.clone();
        push_unary_tape_entry(t.id(), out.id(), move |g| {
            binary_op_metal(g, &t_capture, "clamp_grad", move |grad, x| {
                if x < min_f || x > max_f { 0.0 } else { grad }
            })
        });
        Ok(out)
    }

    // ── Broadcast binaries ─────────────────────────────────────────────────

    /// `atan2(y, x)` with CPU's argument order (lhs is y). Quotient rule:
    /// `d/dy = g * x / (x^2 + y^2)`, `d/dx = g * (-y) / (x^2 + y^2)`.
    pub(crate) fn atan2<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = binary_op_metal(lhs, rhs, "atan2", |y, x| y.atan2(x))?;
        let (y_cap, x_cap) = (lhs.clone(), rhs.clone());
        let (y_dims, x_dims) = (
            lhs.metadata().shape().dims().to_vec(),
            rhs.metadata().shape().dims().to_vec(),
        );
        let (y_id, x_id, out_id) = (lhs.id(), rhs.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![y_id, x_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                let x_sq = binary_op_metal(&x_cap, &x_cap, "atan2_grad_x_sq", |a, b| a * b)?;
                let y_sq = binary_op_metal(&y_cap, &y_cap, "atan2_grad_y_sq", |a, b| a * b)?;
                let denom = binary_op_metal(&x_sq, &y_sq, "atan2_grad_denom", |a, b| a + b)?;
                let numer_y =
                    binary_op_metal(grad_out, &x_cap, "atan2_grad_numer_y", |a, b| a * b)?;
                let grad_y = binary_op_metal(&numer_y, &denom, "atan2_grad_y", |a, b| a / b)?;
                let neg_y = unary_op_metal(&y_cap, |x| -x)?;
                let numer_x =
                    binary_op_metal(grad_out, &neg_y, "atan2_grad_numer_x", |a, b| a * b)?;
                let grad_x = binary_op_metal(&numer_x, &denom, "atan2_grad_x", |a, b| a / b)?;
                Ok(vec![
                    unbroadcast(&grad_y, &y_dims)?,
                    unbroadcast(&grad_x, &x_dims)?,
                ])
            }),
        });
        Ok(out)
    }

    /// `fmod(lhs, rhs) = a % b` (truncated remainder, CPU `canonical_fmod`).
    /// Shares [`push_modulus_tape`](Self::push_modulus_tape) with
    /// [`remainder`](Self::remainder).
    pub(crate) fn fmod<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = binary_op_metal(lhs, rhs, "fmod", |a, b| a % b)?;
        Self::push_modulus_tape::<K>(lhs, rhs, &out);
        Ok(out)
    }

    /// `remainder(lhs, rhs) = a.rem_euclid(b)` (CPU `canonical_remainder`).
    /// Same backward as [`fmod`](Self::fmod); only the forward rounding
    /// convention differs, and `q` is recovered from the output.
    pub(crate) fn remainder<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = binary_op_metal(lhs, rhs, "remainder", |a, b| a.rem_euclid(b))?;
        Self::push_modulus_tape::<K>(lhs, rhs, &out);
        Ok(out)
    }

    /// `dropout(t, probability, training)`: identity when eval or
    /// `p <= 0`, zeroed when `p >= 1` (the descriptor only admits
    /// `[0, 1)`, so the branch is defensive), otherwise a counter-based
    /// keep-mask scaled by `1 / (1 - p)` — CPU's and CUDA's (#84) recipe,
    /// with the same `hash_uniform(seed, index)` mix so the draw is
    /// reproducible from the process seed and a flat index. The mask is
    /// constant; the tape rides `mul` and `mul_scalar_float`, so the
    /// gradient multiplies by the same mask. The identity path returns the
    /// operand itself (same tensor id), so a gradient arriving there needs
    /// no entry of its own — the clone-links-identity pattern CPU and CUDA
    /// use.
    pub(crate) fn dropout<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        probability: f64,
        training: bool,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        if !training || probability <= 0.0 {
            return Ok(t.clone());
        }
        if probability >= 1.0 {
            return Self::mul_scalar_float::<K>(t, 0.0);
        }
        let dims = t.metadata().shape().dims().to_vec();
        let numel = crate::bytes::checked_numel(&dims)?;
        let numel_u64 = u64::try_from(numel).map_err(|_| {
            incin_core::error::Error::Msg("dropout mask element count exceeds u64".into())
        })?;
        let (seed, start) = reserve_draw(numel_u64);
        let p = probability as f32;
        // `step(draw - p)`: keep when `draw > p`, matching CPU's
        // `canonical_step` on the shifted draw — materialized directly
        // rather than through `add_scalar`/`step` because the draws have no
        // producer on the tape either way, so those two entries would be
        // dead weight.
        let mask_data: Vec<f32> = (0..numel_u64)
            .map(|i| {
                if hash_uniform(seed, start.wrapping_add(i)) > p {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let mask = storage_from_f32(&mask_data, &dims, t)?;
        let kept = Self::mul::<K>(t, &mask)?;
        Self::mul_scalar_float::<K>(&kept, 1.0 / (1.0 - probability))
    }

    /// Shared `record_modulus` tape entry for `fmod`/`remainder`.
    ///
    /// `r = a - b * q` with `q` locally constant, so `dr/da = 1` and
    /// `dr/db = -q`, with `q = (a - r) / b` recovered from the values so
    /// neither recipe restates a rounding rule that could drift from its
    /// forward (CPU `record_modulus`).
    fn push_modulus_tape<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
        out: &<Self as StorageBackend>::Storage<K>,
    ) {
        let (lhs_cap, rhs_cap, out_cap) = (lhs.clone(), rhs.clone(), out.clone());
        let (lhs_dims, rhs_dims) = (
            lhs.metadata().shape().dims().to_vec(),
            rhs.metadata().shape().dims().to_vec(),
        );
        let (lhs_id, rhs_id, out_id) = (lhs.id(), rhs.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                let numerator = binary_op_metal(&lhs_cap, &out_cap, "mod_grad_num", |a, r| a - r)?;
                let quotient = binary_op_metal(&numerator, &rhs_cap, "mod_grad_q", |n, b| n / b)?;
                let grad_rhs = binary_op_metal(grad_out, &quotient, "mod_grad_rhs", |g, q| g * -q)?;
                Ok(vec![
                    unbroadcast(grad_out, &lhs_dims)?,
                    unbroadcast(&grad_rhs, &rhs_dims)?,
                ])
            }),
        });
    }
}

#[cfg(test)]
/// Host-side forward/backward parity tests for the Batch-A elementwise set.
/// Pure `Vec<f32>` math, so they run without a Metal device.
mod tests {
    use super::*;
    use incin_core::exec::GradMode;
    use incin_core::shapes::ShapeBuf;
    use incin_core::tensor::device::{DeviceId, Metal};
    use incin_core::tensor::dtype::DTypeId;

    use crate::metal::storage::MetalStorageMode;
    use crate::metal::tape::MetalGrads;

    type B = MetalBackendImpl<Metal>;

    fn storage(values: &[f32], shape: &[usize]) -> MetalStorage {
        let bytes: Vec<u8> = bytemuck::cast_slice(values).to_vec();
        let meta = incin_core::exec::TensorMeta::contiguous(
            ShapeBuf::from_slice(shape),
            DTypeId::F32.into(),
            DeviceId::metal(0),
            MetalStorage::alignment(),
            values.len(),
        )
        .expect("contiguous metadata for test storage");
        MetalStorage::from_bytes(bytes, meta, MetalStorageMode::Shared, 0)
            .expect("bytes cover the metadata span")
    }

    fn vector(v: &[f32]) -> MetalStorage {
        storage(v, &[v.len()])
    }

    fn read(s: &MetalStorage) -> Vec<f32> {
        bytemuck::cast_slice(s.as_bytes().expect("shared-mode storage is host-readable")).to_vec()
    }

    fn assert_close(got: &[f32], want: &[f32], eps: f32) {
        assert_eq!(
            got.len(),
            want.len(),
            "length mismatch: {got:?} vs {want:?}"
        );
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            // Domain edges: `log`/`log2`/`log10` of `0.0` are `-inf` on
            // both sides (and negatives are `NaN` on both). CPU evaluates
            // the same IEEE edge cases, so matching non-finite values are
            // equal — `(-inf) - (-inf)` is `NaN` and would fail the
            // tolerance check below for no semantic reason.
            if g.is_nan() && w.is_nan() {
                continue;
            }
            if *g == *w {
                continue;
            }
            let tol = eps * w.abs().max(1.0);
            assert!(
                (g - w).abs() <= tol,
                "index {i}: got {g}, want {w} (eps {eps})"
            );
        }
    }

    /// Run `f` under recording mode and return `(output, grads, depth_delta)`.
    fn recorded<F>(f: F) -> (MetalStorage, MetalGrads, usize)
    where
        F: FnOnce() -> MetalStorage,
    {
        let before = crate::metal::tape::depth();
        let out = GradMode::Enabled.scope(f);
        let grads = crate::metal::tape::backward(&out).expect("backward walk succeeds");
        let depth = crate::metal::tape::depth();
        (out, grads, before.saturating_sub(depth))
    }

    // ── Forward parity (table-driven) ──────────────────────────────────────

    #[test]
    fn unary_forwards_match_cpu_eval_f32() {
        // Inputs chosen to hit every branch: positive, negative, zero,
        // fractional, and values inside/outside `clamp`'s bounds.
        let inputs: &[f32] = &[-1.5, -0.25, 0.0, 0.5, 1.0, 2.5, 7.0];

        struct Case {
            name: &'static str,
            run: fn(&[f32]) -> Vec<f32>,
            want: fn(f32) -> f32,
        }

        let cases: &[Case] = &[
            Case {
                name: "relu",
                run: |v| read(&B::relu::<f32>(&vector(v)).unwrap()),
                want: |x| x.max(0.0),
            },
            Case {
                name: "step",
                run: |v| read(&B::step::<f32>(&vector(v)).unwrap()),
                want: step_f32,
            },
            Case {
                name: "abs",
                run: |v| read(&B::abs::<f32>(&vector(v)).unwrap()),
                want: |x| x.abs(),
            },
            Case {
                name: "neg",
                run: |v| read(&B::neg::<f32>(&vector(v)).unwrap()),
                want: |x| -x,
            },
            Case {
                name: "exp",
                run: |v| read(&B::exp::<f32>(&vector(v)).unwrap()),
                want: |x| x.exp(),
            },
            Case {
                // CPU's `UnaryOp::Sqrt` is `value.sqrt()`, which is `NaN`
                // for a negative input — match that, not `abs().sqrt()`.
                name: "sqrt",
                run: |v| read(&B::sqrt::<f32>(&vector(v)).unwrap()),
                want: |x| x.sqrt(),
            },
            Case {
                name: "sign",
                run: |v| read(&B::sign::<f32>(&vector(v)).unwrap()),
                want: sign_f32,
            },
            Case {
                name: "floor",
                run: |v| read(&B::floor::<f32>(&vector(v)).unwrap()),
                want: |x| x.floor(),
            },
            Case {
                name: "ceil",
                run: |v| read(&B::ceil::<f32>(&vector(v)).unwrap()),
                want: |x| x.ceil(),
            },
            Case {
                name: "round",
                run: |v| read(&B::round::<f32>(&vector(v)).unwrap()),
                want: |x| x.round(),
            },
            Case {
                name: "trunc",
                run: |v| read(&B::trunc::<f32>(&vector(v)).unwrap()),
                want: |x| x.trunc(),
            },
            Case {
                name: "frac",
                run: |v| read(&B::frac::<f32>(&vector(v)).unwrap()),
                want: |x| x.fract(),
            },
            Case {
                name: "sin",
                run: |v| read(&B::sin::<f32>(&vector(v)).unwrap()),
                want: |x| x.sin(),
            },
            Case {
                name: "cos",
                run: |v| read(&B::cos::<f32>(&vector(v)).unwrap()),
                want: |x| x.cos(),
            },
            Case {
                name: "tan",
                run: |v| read(&B::tan::<f32>(&vector(v)).unwrap()),
                want: |x| x.tan(),
            },
            Case {
                name: "sinh",
                run: |v| read(&B::sinh::<f32>(&vector(v)).unwrap()),
                want: |x| x.sinh(),
            },
            Case {
                name: "cosh",
                run: |v| read(&B::cosh::<f32>(&vector(v)).unwrap()),
                want: |x| x.cosh(),
            },
            Case {
                name: "elu",
                run: |v| read(&B::elu::<f32>(&vector(v)).unwrap()),
                want: |x| if x > 0.0 { x } else { x.exp() - 1.0 },
            },
            Case {
                name: "mish",
                run: |v| read(&B::mish::<f32>(&vector(v)).unwrap()),
                want: |x| x * softplus_f32(x).tanh(),
            },
            Case {
                name: "swish",
                run: |v| read(&B::swish::<f32>(&vector(v)).unwrap()),
                want: |x| x / (1.0 + (-x).exp()),
            },
            Case {
                name: "sigmoid",
                run: |v| read(&B::sigmoid::<f32>(&vector(v)).unwrap()),
                want: |x| 1.0 / (1.0 + (-x).exp()),
            },
            Case {
                name: "tanh",
                run: |v| read(&B::tanh::<f32>(&vector(v)).unwrap()),
                want: |x| x.tanh(),
            },
            Case {
                name: "erf",
                run: |v| read(&B::erf::<f32>(&vector(v)).unwrap()),
                want: |x| erf_approx_f64(f64::from(x)) as f32,
            },
            Case {
                name: "log2_pos",
                run: |v| read(&B::log2::<f32>(&vector(v)).unwrap()),
                want: |x| x.log2(),
            },
            Case {
                name: "log10_pos",
                run: |v| read(&B::log10::<f32>(&vector(v)).unwrap()),
                want: |x| x.log10(),
            },
            Case {
                name: "asin_domain",
                run: |v| read(&B::asin::<f32>(&vector(&[v[3]])).unwrap()),
                want: |x| x.asin(),
            },
            Case {
                name: "acos_domain",
                run: |v| read(&B::acos::<f32>(&vector(&[v[3]])).unwrap()),
                want: |x| x.acos(),
            },
            Case {
                name: "atan",
                run: |v| read(&B::atan::<f32>(&vector(v)).unwrap()),
                want: |x| x.atan(),
            },
            Case {
                name: "asinh",
                run: |v| read(&B::asinh::<f32>(&vector(v)).unwrap()),
                want: |x| x.asinh(),
            },
            Case {
                name: "acosh_domain",
                run: |v| read(&B::acosh::<f32>(&vector(&[v[6]])).unwrap()),
                want: |x| x.acosh(),
            },
            Case {
                name: "atanh_domain",
                run: |v| read(&B::atanh::<f32>(&vector(&[v[3]])).unwrap()),
                want: |x| x.atanh(),
            },
            Case {
                name: "rsqrt_pos",
                run: |v| read(&B::rsqrt::<f32>(&vector(&[v[3]])).unwrap()),
                want: |x| 1.0 / x.sqrt(),
            },
            Case {
                name: "log_pos",
                run: |v| read(&B::log::<f32>(&vector(&[v[3]])).unwrap()),
                want: |x| x.ln(),
            },
        ];

        for case in cases {
            let got = (case.run)(inputs);
            let want: Vec<f32> = inputs.iter().map(|&x| (case.want)(x)).collect();
            // The domain-filtered cases return length-1 outputs; only compare
            // when the shapes line up.
            if got.len() == want.len() {
                assert_close(&got, &want, 1e-5);
            } else {
                assert_eq!(got.len(), 1, "{} returned unexpected length", case.name);
            }
        }
    }

    #[test]
    fn gelu_forward_matches_erf_form() {
        let t = vector(&[-2.0, -0.5, 0.0, 0.5, 2.0]);
        let got = read(&B::gelu::<f32>(&t).unwrap());
        let want: Vec<f32> = [-2.0f32, -0.5, 0.0, 0.5, 2.0]
            .iter()
            .map(|&x| {
                let x64 = f64::from(x);
                (x64 * 0.5 * (1.0 + erf_approx_f64(x64 / core::f64::consts::SQRT_2))) as f32
            })
            .collect();
        assert_close(&got, &want, 1e-6);
    }

    #[test]
    fn scalar_forwards_and_powf_clamp_match_cpu() {
        let t = vector(&[-2.0, 0.0, 1.5, 4.0]);

        assert_eq!(
            read(&B::add_scalar_float::<f32>(&t, 1.0).unwrap()),
            vec![-1.0, 1.0, 2.5, 5.0]
        );
        assert_eq!(
            read(&B::sub_scalar_float::<f32>(&t, 1.0).unwrap()),
            vec![-3.0, -1.0, 0.5, 3.0]
        );
        assert_eq!(
            read(&B::mul_scalar_float::<f32>(&t, 2.0).unwrap()),
            vec![-4.0, 0.0, 3.0, 8.0]
        );
        assert_eq!(
            read(&B::div_scalar_float::<f32>(&t, 2.0).unwrap()),
            vec![-1.0, 0.0, 0.75, 2.0]
        );

        // powf: (-2)^3, 0^3, 1.5^3, 4^3 — f32::powf matches CPU's f32 path.
        let pow = read(&B::powf::<f32>(&t, 3.0).unwrap());
        assert_close(&pow, &[-8.0, 0.0, 3.375, 64.0], 1e-5);

        // clamp: bounds as f32, interior passes, ends saturate.
        let clamped = read(&B::clamp::<f32>(&t, -1.0, 2.0).unwrap());
        assert_eq!(clamped, vec![-1.0, 0.0, 1.5, 2.0]);
    }

    #[test]
    fn binary_forwards_broadcast_and_match_cpu() {
        // Same-shape atan2/fmod/remainder.
        let y = vector(&[1.0, -1.0, 0.0, 3.0]);
        let x = vector(&[1.0, 1.0, -1.0, 2.0]);

        let a = read(&B::atan2::<f32>(&y, &x).unwrap());
        assert_close(
            &a,
            &[
                1.0f32.atan2(1.0),
                (-1.0f32).atan2(1.0),
                0.0f32.atan2(-1.0),
                3.0f32.atan2(2.0),
            ],
            1e-6,
        );

        let f =
            read(&B::fmod::<f32>(&vector(&[7.0, -7.0, 7.0]), &vector(&[3.0, 3.0, -3.0])).unwrap());
        // f32 % is truncated remainder: 7%3=1, -7%3=-1, 7%-3=1
        assert_close(&f, &[1.0, -1.0, 1.0], 1e-6);

        let r = read(&B::remainder::<f32>(&vector(&[7.0, -7.0]), &vector(&[3.0, 3.0])).unwrap());
        // rem_euclid: always least non-negative → 1, 2
        assert_close(&r, &[1.0, 2.0], 1e-6);
    }

    #[test]
    fn binary_ops_broadcast_shapes() {
        // [2,1] against [1,3] → [2,3], exercising binary_op_metal's stride path.
        let lhs = storage(&[1.0, 2.0], &[2, 1]);
        let rhs = storage(&[10.0, 20.0, 30.0], &[1, 3]);
        let out = B::atan2::<f32>(&lhs, &rhs).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        let got = read(&out);
        assert_eq!(got.len(), 6);
        // First row (y=1): atan2(1, 10/20/30)
        assert_close(
            &got[0..3],
            &[1.0f32.atan2(10.0), 1.0f32.atan2(20.0), 1.0f32.atan2(30.0)],
            1e-6,
        );
    }

    // ── Backward recipes ───────────────────────────────────────────────────

    #[test]
    fn relu_backward_is_step_mask() {
        let t = vector(&[-1.0, 0.0, 2.0]);
        let (out, grads, _) = recorded(|| B::relu::<f32>(&t).unwrap());
        assert_eq!(read(&out), vec![0.0, 0.0, 2.0]);
        let g = grads.get(t.id()).expect("relu records an input grad");
        // step: x>0 → 1, else 0; seed is ones_like(out)
        assert_eq!(read(g), vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn scalar_backward_recipes_match_cpu() {
        let t = vector(&[2.0, -4.0]);

        let (_, grads, _) = recorded(|| B::add_scalar_float::<f32>(&t, 3.0).unwrap());
        assert_eq!(read(grads.get(t.id()).unwrap()), vec![1.0, 1.0]);

        let (_, grads, _) = recorded(|| B::sub_scalar_float::<f32>(&t, 3.0).unwrap());
        assert_eq!(read(grads.get(t.id()).unwrap()), vec![1.0, 1.0]);

        let (_, grads, _) = recorded(|| B::mul_scalar_float::<f32>(&t, 2.0).unwrap());
        assert_eq!(read(grads.get(t.id()).unwrap()), vec![2.0, 2.0]);

        let (_, grads, _) = recorded(|| B::div_scalar_float::<f32>(&t, 2.0).unwrap());
        assert_eq!(read(grads.get(t.id()).unwrap()), vec![0.5, 0.5]);
    }

    #[test]
    fn powf_backward_is_p_x_to_p_minus_1() {
        let t = vector(&[2.0, 3.0]);
        let (_, grads, _) = recorded(|| B::powf::<f32>(&t, 3.0).unwrap());
        // d/dx x^3 = 3x^2 → 12, 27
        assert_close(
            read(grads.get(t.id()).unwrap()).as_slice(),
            &[12.0, 27.0],
            1e-4,
        );
    }

    #[test]
    fn clamp_backward_stops_outside_bounds_and_passes_inside() {
        let t = vector(&[-2.0, 0.0, 1.0, 3.0]);
        let (_, grads, _) = recorded(|| B::clamp::<f32>(&t, -1.0, 2.0).unwrap());
        // -2 < min → 0; 0 and 1 interior → 1; 3 > max → 0
        assert_eq!(read(grads.get(t.id()).unwrap()), vec![0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn exp_and_sqrt_backward_are_output_based() {
        let t = vector(&[0.0, 1.0]);
        let (out, grads, _) = recorded(|| B::exp::<f32>(&t).unwrap());
        assert_eq!(read(grads.get(t.id()).unwrap()), read(&out));

        let t = vector(&[4.0, 9.0]);
        let (out, grads, _) = recorded(|| B::sqrt::<f32>(&t).unwrap());
        // g=1 → 1/(2*out) = 0.25, 1/6
        assert_close(
            read(grads.get(t.id()).unwrap()).as_slice(),
            &[0.25, 1.0 / 6.0],
            1e-5,
        );
        assert_eq!(read(&out), vec![2.0, 3.0]);
    }

    #[test]
    fn zero_grad_unaries_record_a_zero_cotangent() {
        for (name, run) in [
            ("step", B::step::<f32> as fn(&MetalStorage) -> _),
            ("sign", B::sign::<f32>),
            ("floor", B::floor::<f32>),
            ("ceil", B::ceil::<f32>),
            ("round", B::round::<f32>),
            ("trunc", B::trunc::<f32>),
        ] {
            let t = vector(&[1.5, -2.5]);
            let (_, grads, _) = recorded(|| run(&t).unwrap());
            let g = grads
                .get(t.id())
                .unwrap_or_else(|| panic!("{name} must record a grad"));
            assert_eq!(
                read(g),
                vec![0.0, 0.0],
                "{name} must record an all-zero gradient"
            );
        }
    }

    #[test]
    fn frac_backward_is_identity() {
        let t = vector(&[1.25, -0.5]);
        let (_, grads, _) = recorded(|| B::frac::<f32>(&t).unwrap());
        assert_eq!(read(grads.get(t.id()).unwrap()), vec![1.0, 1.0]);
    }

    #[test]
    fn atan2_backward_matches_quotient_rule() {
        let y = vector(&[1.0, 0.0]);
        let x = vector(&[1.0, 1.0]);
        let (out, grads, _) = recorded(|| B::atan2::<f32>(&y, &x).unwrap());
        // Seed = ones (same shape as out).
        // denom = x^2+y^2 = 2, 1
        // gy = 1 * x / denom = 0.5, 1
        // gx = 1 * (-y) / denom = -0.5, 0
        assert_close(
            read(grads.get(y.id()).unwrap()).as_slice(),
            &[0.5, 1.0],
            1e-5,
        );
        assert_close(
            read(grads.get(x.id()).unwrap()).as_slice(),
            &[-0.5, 0.0],
            1e-5,
        );
        let _ = out;
    }

    #[test]
    fn modulus_backwards_recover_quotient_from_output() {
        // fmod(7, 3) = 1 → q = (7-1)/3 = 2; gy = 1, gx = -q = -2
        let a = vector(&[7.0]);
        let b = vector(&[3.0]);
        let (_, grads, _) = recorded(|| B::fmod::<f32>(&a, &b).unwrap());
        assert_close(read(grads.get(a.id()).unwrap()).as_slice(), &[1.0], 1e-5);
        assert_close(read(grads.get(b.id()).unwrap()).as_slice(), &[-2.0], 1e-5);

        // remainder(-7, 3) = 2 → q = (-7-2)/3 = -3; gx = -q = 3
        let a = vector(&[-7.0]);
        let b = vector(&[3.0]);
        let (_, grads, _) = recorded(|| B::remainder::<f32>(&a, &b).unwrap());
        assert_close(read(grads.get(a.id()).unwrap()).as_slice(), &[1.0], 1e-5);
        assert_close(read(grads.get(b.id()).unwrap()).as_slice(), &[3.0], 1e-5);
    }

    #[test]
    fn nograd_records_nothing() {
        let t = vector(&[1.0, 2.0]);
        let before = crate::metal::tape::depth();
        let _ = GradMode::Disabled.scope(|| B::relu::<f32>(&t).unwrap());
        let _ = GradMode::Disabled.scope(|| B::add_scalar_float::<f32>(&t, 1.0).unwrap());
        let _ = GradMode::Disabled.scope(|| B::atan2::<f32>(&t, &t).unwrap());
        let _ = GradMode::Disabled.scope(|| B::dropout::<f32>(&t, 0.5, true).unwrap());
        assert_eq!(
            crate::metal::tape::depth(),
            before,
            "NoGrad must record nothing"
        );
    }

    #[test]
    fn tanh_sigmoid_backward_are_output_based() {
        let t = vector(&[0.0, 1.0]);
        let (out, grads, _) = recorded(|| B::tanh::<f32>(&t).unwrap());
        let o = read(&out);
        let want: Vec<f32> = o.iter().map(|&v| 1.0 - v * v).collect();
        assert_close(read(grads.get(t.id()).unwrap()).as_slice(), &want, 1e-5);

        let (out, grads, _) = recorded(|| B::sigmoid::<f32>(&t).unwrap());
        let o = read(&out);
        let want: Vec<f32> = o.iter().map(|&v| v * (1.0 - v)).collect();
        assert_close(read(grads.get(t.id()).unwrap()).as_slice(), &want, 1e-5);
    }

    #[test]
    fn abs_backward_uses_sign_mask() {
        let t = vector(&[-2.0, 0.0, 3.0]);
        let (_, grads, _) = recorded(|| B::abs::<f32>(&t).unwrap());
        assert_eq!(read(grads.get(t.id()).unwrap()), vec![-1.0, 0.0, 1.0]);
    }

    // ── dropout ────────────────────────────────────────────────────────────

    #[test]
    fn dropout_outside_training_is_the_operand_itself() {
        let t = vector(&[1.0, -2.0, 3.0, 0.0]);
        let before = crate::metal::tape::depth();
        let out = GradMode::Enabled.scope(|| B::dropout::<f32>(&t, 0.5, false).unwrap());
        assert_eq!(read(&out), read(&t), "eval-mode dropout is the identity");
        assert_eq!(
            out.id(),
            t.id(),
            "the identity path links the gradient by sharing the operand's id"
        );
        assert_eq!(
            crate::metal::tape::depth(),
            before,
            "the identity path pushes no tape entry"
        );

        // p <= 0 is identity even in training mode.
        let zero = GradMode::Enabled.scope(|| B::dropout::<f32>(&t, 0.0, true).unwrap());
        assert_eq!(zero.id(), t.id());
    }

    #[test]
    fn dropout_training_zeroes_or_scales_and_the_gradient_carries_the_mask() {
        let values = [1.0f32, -2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let t = vector(&values);
        // Count the forward's own entries before `backward` drains them.
        let before = crate::metal::tape::depth();
        let out = GradMode::Enabled.scope(|| B::dropout::<f32>(&t, 0.5, true).unwrap());
        let added = crate::metal::tape::depth() - before;
        assert!(added >= 2, "mul + mul_scalar_float record (got {added})");
        let grads = crate::metal::tape::backward(&out).expect("backward walk succeeds");

        let got = read(&out);
        assert_eq!(got.len(), values.len(), "dropout preserves the shape");
        let scale = 1.0 / (1.0 - 0.5); // 2.0
        for (i, (&g, &x)) in got.iter().zip(values.iter()).enumerate() {
            assert!(
                g == 0.0 || (g - x * scale).abs() <= 1e-5,
                "training[{i}]: got {g}, expected 0 or {}",
                x * scale
            );
        }

        // d/dx of `x * mask * scale` is `mask * scale`: 0 where dropped,
        // the keep reciprocal where kept — the same mask the forward drew.
        let grad = read(grads.get(t.id()).expect("dropout records an input grad"));
        for (i, (&gi, &g)) in grad.iter().zip(got.iter()).enumerate() {
            if g == 0.0 {
                assert_eq!(gi, 0.0, "dropped element {i} must not receive gradient");
            } else {
                assert!(
                    (gi - scale).abs() <= 1e-5,
                    "kept element {i}: gradient {gi}, expected {scale}"
                );
            }
        }
    }

    #[test]
    fn dropout_p_at_or_above_one_zeroes() {
        let t = vector(&[1.0, 2.0]);
        let out = B::dropout::<f32>(&t, 1.0, true).unwrap();
        assert_close(&read(&out), &[0.0, 0.0], 0.0);
    }

    #[test]
    fn dropout_hash_uniform_is_deterministic_and_in_the_unit_interval() {
        // Pins the #84 counter mix: same pair is the same draw, adjacent
        // indices differ, and every draw lands in [0, 1).
        assert_eq!(hash_uniform(7, 123), hash_uniform(7, 123));
        assert_ne!(hash_uniform(7, 123), hash_uniform(7, 124));
        assert_ne!(hash_uniform(7, 123), hash_uniform(8, 123));
        for i in 0..10_000u64 {
            let v = hash_uniform(42, i);
            assert!((0.0..1.0).contains(&v), "hash_uniform(42, {i}) = {v}");
        }
        assert_ne!(mix64(0), mix64(1), "adjacent inputs must scatter");
    }

    #[test]
    fn dropout_draws_advance_so_consecutive_calls_do_not_collide() {
        let t = vector(&[1.0, 2.0, 3.0, 4.0]);
        let before = reserve_draw(0).1;
        let _ = B::dropout::<f32>(&t, 0.5, true).unwrap();
        let mid = reserve_draw(0).1;
        assert_eq!(mid, before + 4, "one training draw advances by numel");
        let _ = B::dropout::<f32>(&t, 0.5, true).unwrap();
        let end = reserve_draw(0).1;
        assert_eq!(end, mid + 4, "the second draw starts past the first");
    }
}
