use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum UnaryOp {
    Neg,
    AddScalar(f64),
    MulScalar(f64),
    Relu,
    Step,
    Mish,
    Elu,
    Gelu,
    Abs,
    Exp,
    Sqrt,
    Log,
    Tanh,
    Sigmoid,
    Swish,
    Powf(f64),
    Clamp(f64, f64),
    Sign,
    Floor,
    Ceil,
    Round,
    Log2,
    Log10,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Asinh,
    Acosh,
    Atanh,
    Erf,
    Rsqrt,
    Trunc,
    Frac,
    TanhBackward,
    SigmoidBackward,
    EluBackward,
    TanBackward,
    AsinBackward,
    AcosBackward,
    AtanBackward,
    AsinhBackward,
    AcoshBackward,
    AtanhBackward,
    ErfBackward,
    RsqrtBackward,
    GeluBackward,
    MishBackward,
}

impl UnaryOp {
    #[inline]
    pub(crate) fn eval_f32(self, value: f32) -> f32 {
        match self {
            Self::Neg => -value,
            Self::AddScalar(scalar) => value + scalar as f32,
            Self::MulScalar(scalar) => value * scalar as f32,
            Self::Relu => value.max(0.0),
            Self::Step => {
                if value > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Mish => {
                let softplus = if value > 20.0 {
                    value
                } else {
                    (1.0 + value.exp()).ln()
                };
                value * softplus.tanh()
            }
            Self::Elu => {
                if value > 0.0 {
                    value
                } else {
                    value.exp() - 1.0
                }
            }
            Self::Gelu => {
                // GELU's analytical backward uses the normal PDF/CDF in F64.
                // Keeping this approximation in F64 avoids exceeding the
                // established finite-difference gradient tolerance.
                let value = f64::from(value);
                (value * 0.5 * (1.0 + erf_approx_f64(value / core::f64::consts::SQRT_2))) as f32
            }
            Self::Abs => value.abs(),
            Self::Exp => value.exp(),
            Self::Sqrt => value.sqrt(),
            Self::Log => value.ln(),
            Self::Tanh => value.tanh(),
            Self::Sigmoid => 1.0 / (1.0 + (-value).exp()),
            Self::Swish => value / (1.0 + (-value).exp()),
            Self::Powf(exp) => value.powf(exp as f32),
            Self::Clamp(min, max) => value.clamp(min as f32, max as f32),
            Self::Sign => {
                if value > 0.0 {
                    1.0
                } else if value < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
            Self::Floor => value.floor(),
            Self::Ceil => value.ceil(),
            Self::Round => value.round(),
            Self::Log2 => value.log2(),
            Self::Log10 => value.log10(),
            Self::Sin => value.sin(),
            Self::Cos => value.cos(),
            Self::Tan => value.tan(),
            Self::Asin => value.asin(),
            Self::Acos => value.acos(),
            Self::Atan => value.atan(),
            Self::Sinh => value.sinh(),
            Self::Cosh => value.cosh(),
            Self::Asinh => value.asinh(),
            Self::Acosh => value.acosh(),
            Self::Atanh => value.atanh(),
            Self::Erf => erf_approx_f64(f64::from(value)) as f32,
            Self::Rsqrt => 1.0 / value.sqrt(),
            Self::Trunc => value.trunc(),
            Self::Frac => value.fract(),
            Self::TanhBackward => 1.0 - value * value,
            Self::SigmoidBackward => value * (1.0 - value),
            Self::EluBackward => {
                if value > 0.0 {
                    1.0
                } else {
                    value + 1.0
                }
            }
            Self::TanBackward => 1.0 + value.tan().powi(2),
            Self::AsinBackward => 1.0 / (1.0 - value * value).sqrt(),
            Self::AcosBackward => -1.0 / (1.0 - value * value).sqrt(),
            Self::AtanBackward => 1.0 / (1.0 + value * value),
            Self::AsinhBackward => 1.0 / (value * value + 1.0).sqrt(),
            Self::AcoshBackward => 1.0 / (value * value - 1.0).sqrt(),
            Self::AtanhBackward => 1.0 / (1.0 - value * value),
            Self::ErfBackward => (2.0 / core::f32::consts::PI.sqrt()) * (-value * value).exp(),
            Self::RsqrtBackward => -0.5 / (value * value.sqrt()),
            Self::GeluBackward => {
                let value_f64 = f64::from(value);
                let cdf = 0.5 * (1.0 + erf_approx_f64(value_f64 / core::f64::consts::SQRT_2));
                let pdf = (1.0 / (2.0 * core::f64::consts::PI).sqrt())
                    * (-value_f64 * value_f64 / 2.0).exp();
                (cdf + value_f64 * pdf) as f32
            }
            Self::MishBackward => {
                let softplus = if value > 20.0 {
                    value
                } else {
                    (1.0 + value.exp()).ln()
                };
                let tanh = softplus.tanh();
                let sigmoid = 1.0 / (1.0 + (-value).exp());
                tanh + value * sigmoid * (1.0 - tanh * tanh)
            }
        }
    }

    #[inline]
    pub(crate) fn eval_f64(self, value: f64) -> f64 {
        match self {
            Self::Neg => -value,
            Self::AddScalar(scalar) => value + scalar,
            Self::MulScalar(scalar) => value * scalar,
            Self::Relu => value.max(0.0),
            Self::Step => {
                if value > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Mish => {
                let softplus = if value > 20.0 {
                    value
                } else {
                    (1.0 + value.exp()).ln()
                };
                value * softplus.tanh()
            }
            Self::Elu => {
                if value > 0.0 {
                    value
                } else {
                    value.exp() - 1.0
                }
            }
            Self::Gelu => value * 0.5 * (1.0 + erf_approx_f64(value / core::f64::consts::SQRT_2)),
            Self::Abs => value.abs(),
            Self::Exp => value.exp(),
            Self::Sqrt => value.sqrt(),
            Self::Log => value.ln(),
            Self::Tanh => value.tanh(),
            Self::Sigmoid => 1.0 / (1.0 + (-value).exp()),
            Self::Swish => value / (1.0 + (-value).exp()),
            Self::Powf(exp) => value.powf(exp),
            Self::Clamp(min, max) => value.clamp(min, max),
            Self::Sign => {
                if value > 0.0 {
                    1.0
                } else if value < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
            Self::Floor => value.floor(),
            Self::Ceil => value.ceil(),
            Self::Round => value.round(),
            Self::Log2 => value.log2(),
            Self::Log10 => value.log10(),
            Self::Sin => value.sin(),
            Self::Cos => value.cos(),
            Self::Tan => value.tan(),
            Self::Asin => value.asin(),
            Self::Acos => value.acos(),
            Self::Atan => value.atan(),
            Self::Sinh => value.sinh(),
            Self::Cosh => value.cosh(),
            Self::Asinh => value.asinh(),
            Self::Acosh => value.acosh(),
            Self::Atanh => value.atanh(),
            Self::Erf => erf_approx_f64(value),
            Self::Rsqrt => 1.0 / value.sqrt(),
            Self::Trunc => value.trunc(),
            Self::Frac => value.fract(),
            Self::TanhBackward => 1.0 - value * value,
            Self::SigmoidBackward => value * (1.0 - value),
            Self::EluBackward => {
                if value > 0.0 {
                    1.0
                } else {
                    value + 1.0
                }
            }
            Self::TanBackward => 1.0 + value.tan().powi(2),
            Self::AsinBackward => 1.0 / (1.0 - value * value).sqrt(),
            Self::AcosBackward => -1.0 / (1.0 - value * value).sqrt(),
            Self::AtanBackward => 1.0 / (1.0 + value * value),
            Self::AsinhBackward => 1.0 / (value * value + 1.0).sqrt(),
            Self::AcoshBackward => 1.0 / (value * value - 1.0).sqrt(),
            Self::AtanhBackward => 1.0 / (1.0 - value * value),
            Self::ErfBackward => (2.0 / core::f64::consts::PI.sqrt()) * (-value * value).exp(),
            Self::RsqrtBackward => -0.5 / (value * value.sqrt()),
            Self::GeluBackward => {
                let cdf = 0.5 * (1.0 + erf_approx_f64(value / core::f64::consts::SQRT_2));
                let pdf =
                    (1.0 / (2.0 * core::f64::consts::PI).sqrt()) * (-value * value / 2.0).exp();
                cdf + value * pdf
            }
            Self::MishBackward => {
                let softplus = if value > 20.0 {
                    value
                } else {
                    (1.0 + value.exp()).ln()
                };
                let tanh = softplus.tanh();
                let sigmoid = 1.0 / (1.0 + (-value).exp());
                tanh + value * sigmoid * (1.0 - tanh * tanh)
            }
        }
    }
}

impl BinaryOp {
    #[inline]
    pub(crate) fn eval_f32(self, lhs: f32, rhs: f32) -> f32 {
        match self {
            Self::Add => lhs + rhs,
            Self::Sub => lhs - rhs,
            Self::Mul => lhs * rhs,
            Self::Div => lhs / rhs,
        }
    }

    #[inline]
    pub(crate) fn eval_f64(self, lhs: f64, rhs: f64) -> f64 {
        match self {
            Self::Add => lhs + rhs,
            Self::Sub => lhs - rhs,
            Self::Mul => lhs * rhs,
            Self::Div => lhs / rhs,
        }
    }
}

/// Whether the AVX2 `f32` kernels may be called.
///
/// The decision lives here rather than inline at each call so that there is one
/// place to change and one thing to test. It was inline, and it read
/// `simd_lanes::<f32>() >= 8` alone - true only when the compiler was told to
/// assume AVX2, which a stock `cargo build` never is. See
/// `simd::avx2_detected`.
#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[inline]
pub(crate) fn avx2_f32_available() -> bool {
    const LANES: usize = crate::simd::simd_lanes::<f32>();
    LANES >= 8 || crate::simd::avx2_detected()
}

/// [`avx2_f32_available`] for `f64`, where an AVX2 register holds four lanes.
#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[inline]
pub(crate) fn avx2_f64_available() -> bool {
    const LANES: usize = crate::simd::simd_lanes::<f64>();
    LANES >= 4 || crate::simd::avx2_detected()
}
