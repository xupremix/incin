use crate::err::Error;
use crate::err::Result;

/// Fan geometry for initialization calculations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fan {
    /// Fan-in dimension.
    pub fan_in: usize,
    /// Fan-out dimension.
    pub fan_out: usize,
}

impl Fan {
    /// Creates a new Fan struct.
    pub const fn new(fan_in: usize, fan_out: usize) -> Self {
        Self { fan_in, fan_out }
    }
}

/// Semantic role of a parameter during initialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParameterRole {
    /// Layer weight matrix.
    Weight,
    /// Layer bias vector.
    Bias,
    /// Layer scale parameter (e.g. LayerNorm weight).
    Scale,
    /// Layer offset parameter (e.g. LayerNorm bias).
    Offset,
    /// Other unclassified parameter.
    Other,
}

/// Context provided by a layer when lowering a semantic initializer policy.
#[derive(Debug, Clone, Copy)]
pub struct InitContext {
    /// Semantic parameter role.
    pub role: ParameterRole,
    /// Fan geometry, if applicable.
    pub fan: Option<Fan>,
    /// Seed for host-sampled initializers ([`InitPlan::Orthogonal`] and
    /// [`InitPlan::TruncatedNormal`]). The backend-sampled plans (`Uniform`,
    /// `Normal`) draw from device RNG and ignore this. When `None`, lowering
    /// uses [`DEFAULT_HOST_INIT_SEED`], so parameters lowered from the same
    /// seed share identical fills — vary it per parameter.
    pub seed: Option<u64>,
}

impl InitContext {
    /// Creates a new `InitContext` with the specified role and no fan info.
    pub const fn new(role: ParameterRole) -> Self {
        Self {
            role,
            fan: None,
            seed: None,
        }
    }

    /// Attaches fan geometry to this context.
    pub const fn with_fan(mut self, fan_in: usize, fan_out: usize) -> Self {
        self.fan = Some(Fan { fan_in, fan_out });
        self
    }

    /// Attaches a host-sampling seed to this context.
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Default seed for host-sampled initializer plans when the [`InitContext`]
/// carries none. The fills stay deterministic; override per parameter with
/// [`InitContext::with_seed`].
pub const DEFAULT_HOST_INIT_SEED: u64 = 0;

/// Primitive execution plan for parameter initialization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InitPlan {
    /// Fill with zeros.
    Zeros,
    /// Fill with ones.
    Ones,
    /// Fill with constant scalar value.
    Constant(f64),
    /// Fill with uniform random numbers in `[low, high)`.
    Uniform {
        /// Lower bound.
        low: f64,
        /// Upper bound.
        high: f64,
    },
    /// Fill with normal random numbers $N(\text{mean}, \text{std}^2)$.
    Normal {
        /// Mean.
        mean: f64,
        /// Standard deviation.
        std: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
/// Public initializer policy specification.
pub enum Init {
    #[default]
    /// Fill with zeros.
    Zeros,
    /// Fill with ones.
    Ones,
    /// Uniform random numbers in `[0, 1)`.
    Rand,
    /// Standard normal random numbers $N(0, 1)$.
    Randn,
    /// Constant scalar value.
    Constant(f64),
    /// Uniform random numbers in `[-bound, bound]`.
    Uniform {
        /// Absolute bound limit.
        bound: f64,
    },
    /// Kaiming (He) uniform initialization.
    KaimingUniform {
        /// Non-linear gain parameter `a` (default: $\sqrt{5}$).
        a: f64,
    },
    /// Kaiming (He) normal initialization.
    KaimingNormal {
        /// Non-linear gain parameter `a` (default: $\sqrt{5}$).
        a: f64,
    },
    /// Xavier (Glorot) uniform initialization.
    XavierUniform,
    /// Xavier (Glorot) normal initialization.
    XavierNormal,
    /// Orthogonal initialization with gain `1.0`, matching
    /// `torch.nn.init.orthogonal_`.
    Orthogonal {
        /// Multiplicative gain applied after orthonormalization.
        gain: f64,
    },
    /// Truncated-normal initialization, matching `torch.nn.init.trunc_normal_`.
    TruncatedNormal {
        /// Mean of the underlying normal before truncation.
        mean: f64,
        /// Standard deviation of the underlying normal before truncation.
        std: f64,
        /// Lower truncation bound (inclusive).
        a: f64,
        /// Upper truncation bound (inclusive).
        b: f64,
    },
}

impl Init {
    /// Lowers this semantic `Init` policy into a primitive `InitPlan` using the provided `InitContext`.
    pub fn plan(self, context: InitContext) -> Result<InitPlan> {
        match self {
            Init::Zeros => Ok(InitPlan::Zeros),
            Init::Ones => Ok(InitPlan::Ones),
            Init::Rand => Ok(InitPlan::Uniform {
                low: 0.0,
                high: 1.0,
            }),
            Init::Randn => Ok(InitPlan::Normal {
                mean: 0.0,
                std: 1.0,
            }),
            Init::Constant(val) => Ok(InitPlan::Constant(val)),
            Init::Uniform { bound } => Ok(InitPlan::Uniform {
                low: -bound,
                high: bound,
            }),
            Init::KaimingUniform { a } => {
                let fan = context.fan.ok_or_else(|| Error::ShapeMismatch {
                    op: "Init::plan",
                    expected: vec![],
                    got: vec![],
                    msg: alloc::string::String::from("KaimingUniform requires fan context"),
                })?;
                let fan_val = fan.fan_in;
                let std = f64::sqrt(2.0 / ((1.0 + a * a) * fan_val as f64));
                let bound = f64::sqrt(3.0) * std;
                Ok(InitPlan::Uniform {
                    low: -bound,
                    high: bound,
                })
            }
            Init::KaimingNormal { a } => {
                let fan = context.fan.ok_or_else(|| Error::ShapeMismatch {
                    op: "Init::plan",
                    expected: vec![],
                    got: vec![],
                    msg: alloc::string::String::from("KaimingNormal requires fan context"),
                })?;
                let std = f64::sqrt(2.0 / ((1.0 + a * a) * fan.fan_in as f64));
                Ok(InitPlan::Normal { mean: 0.0, std })
            }
            Init::XavierUniform => {
                let fan = context.fan.ok_or_else(|| Error::ShapeMismatch {
                    op: "Init::plan",
                    expected: vec![],
                    got: vec![],
                    msg: alloc::string::String::from("XavierUniform requires fan context"),
                })?;
                let bound = f64::sqrt(6.0 / (fan.fan_in as f64 + fan.fan_out as f64));
                Ok(InitPlan::Uniform {
                    low: -bound,
                    high: bound,
                })
            }
            Init::XavierNormal => {
                let fan = context.fan.ok_or_else(|| Error::ShapeMismatch {
                    op: "Init::plan",
                    expected: vec![],
                    got: vec![],
                    msg: alloc::string::String::from("XavierNormal requires fan context"),
                })?;
                let std = f64::sqrt(2.0 / (fan.fan_in as f64 + fan.fan_out as f64));
                Ok(InitPlan::Normal { mean: 0.0, std })
            }
            Init::Orthogonal { gain } => {
                if !gain.is_finite() {
                    return Err(Error::ShapeMismatch {
                        op: "Init::plan",
                        expected: vec![],
                        got: vec![],
                        msg: alloc::string::String::from("Orthogonal gain must be finite"),
                    });
                }
                // Host-sampled: no primitive plan can express orthonormal
                // values. Sample with `orthogonal_fill` and write the result
                // through the `ParameterInit` host-fill arms (see the
                // integration snippet in the task report); until those land,
                // lowering reports the gap instead of inventing values.
                Err(Error::ShapeMismatch {
                    op: "Init::plan",
                    expected: vec![],
                    got: vec![],
                    msg: alloc::string::String::from(
                        "Orthogonal needs host sampling via orthogonal_fill; \
                         backend execution requires the ParameterInit host-fill arms",
                    ),
                })
            }
            Init::TruncatedNormal { mean, std, a, b } => {
                check_truncated_normal_params(mean, std, a, b)?;
                // Host-sampled, same story as Orthogonal above.
                Err(Error::ShapeMismatch {
                    op: "Init::plan",
                    expected: vec![],
                    got: vec![],
                    msg: alloc::string::String::from(
                        "TruncatedNormal needs host sampling via truncated_normal_fill; \
                         backend execution requires the ParameterInit host-fill arms",
                    ),
                })
            }
        }
    }
}

/// Convenience constructors for initialization policies.
pub fn zeros() -> Init {
    Init::Zeros
}

/// Fill with ones.
pub fn ones() -> Init {
    Init::Ones
}

/// Uniform random in `[0, 1)`.
pub fn rand() -> Init {
    Init::Rand
}

/// Standard normal random.
pub fn randn() -> Init {
    Init::Randn
}

/// Standard normal random alias.
pub fn normal() -> Init {
    Init::Randn
}

/// Fill with constant value.
pub fn constant(value: f64) -> Init {
    Init::Constant(value)
}

/// Uniform random in `[-bound, bound]`.
pub fn uniform(bound: f64) -> Init {
    Init::Uniform { bound }
}

/// Kaiming uniform initialization with default $a = \sqrt{5}$.
pub fn kaiming_uniform() -> Init {
    Init::KaimingUniform { a: f64::sqrt(5.0) }
}

/// Kaiming uniform initialization with explicit $a$.
pub fn kaiming_uniform_with_a(a: f64) -> Init {
    Init::KaimingUniform { a }
}

/// Kaiming normal initialization with default $a = \sqrt{5}$.
pub fn kaiming_normal() -> Init {
    Init::KaimingNormal { a: f64::sqrt(5.0) }
}

/// Kaiming normal initialization with explicit $a$.
pub fn kaiming_normal_with_a(a: f64) -> Init {
    Init::KaimingNormal { a }
}

/// Xavier uniform initialization.
pub fn xavier_uniform() -> Init {
    Init::XavierUniform
}

/// Xavier normal initialization.
pub fn xavier_normal() -> Init {
    Init::XavierNormal
}

/// Orthogonal initialization with gain `1.0`, matching
/// `torch.nn.init.orthogonal_`. The host-sampling seed travels in the
/// [`InitContext`] (see [`InitContext::with_seed`]).
pub fn orthogonal_() -> Init {
    Init::Orthogonal { gain: 1.0 }
}

/// Orthogonal initialization with an explicit gain.
pub fn orthogonal_with_gain(gain: f64) -> Init {
    Init::Orthogonal { gain }
}

/// Truncated-normal initialization with the `torch.nn.init.trunc_normal_`
/// defaults (`mean = 0`, `std = 1`, `a = -2`, `b = 2`). The host-sampling
/// seed travels in the [`InitContext`] (see [`InitContext::with_seed`]).
pub fn trunc_normal_() -> Init {
    Init::TruncatedNormal {
        mean: 0.0,
        std: 1.0,
        a: -2.0,
        b: 2.0,
    }
}

/// Truncated-normal initialization with explicit parameters.
pub fn trunc_normal_with(mean: f64, std: f64, a: f64, b: f64) -> Init {
    Init::TruncatedNormal { mean, std, a, b }
}

/// Deterministic SplitMix64 generator for host-side initialization math.
///
/// `no_std`-compatible: the crate has no RNG dependency, so the host fills
/// below roll their own seeding from an explicit `u64`. Same seed, same
/// stream — which is what makes the initializer unit tests exact.
#[derive(Debug, Clone)]
pub struct InitRng(u64);

impl InitRng {
    /// Creates a generator from an explicit seed.
    pub fn from_seed(seed: u64) -> Self {
        Self(seed)
    }

    /// Next `u64` of the SplitMix64 stream.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Next `f64` uniform in `[0, 1)` (top 53 bits of the stream).
    pub fn next_f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / 9_007_199_254_740_992.0;
        ((self.next_u64() >> 11) as f64) * SCALE
    }

    /// Next standard-normal sample via the Marsaglia polar method (no
    /// trigonometric calls, hence `no_std`-safe on `core` float methods).
    pub fn next_normal(&mut self) -> f64 {
        loop {
            let u = 2.0 * self.next_f64() - 1.0;
            let v = 2.0 * self.next_f64() - 1.0;
            let s = u * u + v * v;
            if s < 1.0 && s > 0.0 {
                return u * f64::sqrt(-2.0 * f64::ln(s) / s);
            }
        }
    }
}

/// Validates truncated-normal parameters, shared by [`truncated_normal_fill`]
/// and [`Init::plan`].
fn check_truncated_normal_params(mean: f64, std: f64, a: f64, b: f64) -> Result<()> {
    if !mean.is_finite() || !std.is_finite() || !a.is_finite() || !b.is_finite() {
        return Err(Error::ShapeMismatch {
            op: "trunc_normal",
            expected: vec![],
            got: vec![],
            msg: alloc::string::String::from("truncated-normal parameters must all be finite"),
        });
    }
    if std <= 0.0 {
        return Err(Error::ShapeMismatch {
            op: "trunc_normal",
            expected: vec![],
            got: vec![],
            msg: alloc::string::String::from("truncated-normal std must be positive"),
        });
    }
    if a >= b {
        return Err(Error::ShapeMismatch {
            op: "trunc_normal",
            expected: vec![],
            got: vec![],
            msg: alloc::string::String::from("truncated-normal requires a < b"),
        });
    }
    Ok(())
}

/// Fills `values` (row-major `rows` x `cols`) with a gain-scaled orthogonal
/// matrix, matching `torch.nn.init.orthogonal_`.
///
/// The fill draws a standard-normal matrix from `rng` and orthonormalizes it
/// with modified Gram-Schmidt in `f64`: columns when `rows >= cols` (so
/// `Q^T Q = I`), rows when `rows < cols` (so `Q Q^T = I`). Rectangular
/// shapes are supported in both directions. A near-zero pivot (probability
/// zero for continuous draws) is resampled from `rng` instead of dividing.
///
/// # Examples
/// ```rust
/// # extern crate incin_core as incin;
/// use incin::nn::init::{InitRng, orthogonal_fill};
/// let mut rng = InitRng::from_seed(7);
/// let mut values = vec![0.0; 4 * 3];
/// orthogonal_fill(&mut values, 4, 3, 1.0, &mut rng).unwrap();
/// // Tall Q has orthonormal columns: Q^T Q is the 3x3 identity.
/// for i in 0..3 {
///     for j in 0..3 {
///         let mut dot = 0.0;
///         for r in 0..4 {
///             dot += values[r * 3 + i] * values[r * 3 + j];
///         }
///         let expected = if i == j { 1.0 } else { 0.0 };
///         assert!((dot - expected).abs() < 1e-9);
///     }
/// }
/// ```
pub fn orthogonal_fill(
    values: &mut [f64],
    rows: usize,
    cols: usize,
    gain: f64,
    rng: &mut InitRng,
) -> Result<()> {
    if rows == 0 || cols == 0 {
        return Err(Error::ShapeMismatch {
            op: "orthogonal_fill",
            expected: vec![],
            got: vec![],
            msg: alloc::string::String::from("orthogonal_fill needs non-zero rows and cols"),
        });
    }
    let total = rows.checked_mul(cols).ok_or_else(|| Error::ShapeMismatch {
        op: "orthogonal_fill",
        expected: vec![],
        got: vec![],
        msg: alloc::string::String::from("orthogonal_fill rows*cols overflows usize"),
    })?;
    if values.len() != total {
        return Err(Error::ShapeMismatch {
            op: "orthogonal_fill",
            expected: alloc::vec![total],
            got: alloc::vec![values.len()],
            msg: alloc::string::String::from("orthogonal_fill slice length must equal rows*cols"),
        });
    }
    if !gain.is_finite() {
        return Err(Error::ShapeMismatch {
            op: "orthogonal_fill",
            expected: vec![],
            got: vec![],
            msg: alloc::string::String::from("orthogonal_fill gain must be finite"),
        });
    }
    for value in values.iter_mut() {
        *value = rng.next_normal();
    }
    // Orthonormalize over the long axis: columns for tall matrices, rows
    // for wide ones. Index arithmetic stays row-major either way.
    let (outer, inner) = if rows >= cols {
        (cols, rows)
    } else {
        (rows, cols)
    };
    let get = |values: &[f64], o: usize, i: usize| -> f64 {
        if rows >= cols {
            values[i * cols + o]
        } else {
            values[o * cols + i]
        }
    };
    let set = |values: &mut [f64], o: usize, i: usize, v: f64| {
        if rows >= cols {
            values[i * cols + o] = v;
        } else {
            values[o * cols + i] = v;
        }
    };
    for o in 0..outer {
        // Resample the whole vector on a near-zero pivot rather than
        // dividing by it; restarts the current outer index.
        let mut attempts = 0;
        loop {
            // Modified Gram-Schmidt: each previous orthonormal vector is
            // projected out of the whole current vector before the next.
            for p in 0..o {
                let mut dot = 0.0;
                for k in 0..inner {
                    dot += get(values, o, k) * get(values, p, k);
                }
                for k in 0..inner {
                    let v = get(values, o, k) - dot * get(values, p, k);
                    set(values, o, k, v);
                }
            }
            let mut norm_sq = 0.0;
            for i in 0..inner {
                let v = get(values, o, i);
                norm_sq += v * v;
            }
            let norm = f64::sqrt(norm_sq);
            if norm > 1e-12 {
                // Normalize to unit length here; the gain is applied once
                // to the whole matrix below. Scaling each vector as it is
                // produced would make later projections subtract gain²
                // times too much (they project against already-scaled
                // vectors), destroying orthogonality for any gain != 1.
                for i in 0..inner {
                    set(values, o, i, get(values, o, i) / norm);
                }
                break;
            }
            attempts += 1;
            if attempts > 100 {
                return Err(Error::ShapeMismatch {
                    op: "orthogonal_fill",
                    expected: vec![],
                    got: vec![],
                    msg: alloc::string::String::from(
                        "orthogonal_fill could not find a non-degenerate vector",
                    ),
                });
            }
            for i in 0..inner {
                set(values, o, i, rng.next_normal());
            }
        }
    }
    if gain != 1.0 {
        for value in values.iter_mut() {
            *value *= gain;
        }
    }
    Ok(())
}

/// Fills `values` with `TruncatedNormal(mean, std, a, b)` samples via
/// rejection: draws `mean + std * N(0, 1)` from `rng` until each draw lands
/// in `[a, b]`. Rejection (rather than an `erf`-based inverse-CDF, which
/// `core` cannot provide `no_std`) keeps the math to `sqrt`/`ln` and makes
/// the bounds exact by construction.
///
/// # Examples
/// ```rust
/// # extern crate incin_core as incin;
/// use incin::nn::init::{InitRng, truncated_normal_fill};
/// let mut rng = InitRng::from_seed(11);
/// let mut values = vec![0.0; 512];
/// truncated_normal_fill(&mut values, 0.0, 1.0, -2.0, 2.0, &mut rng).unwrap();
/// assert!(values.iter().all(|&v| (-2.0..=2.0).contains(&v)));
/// let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
/// assert!(mean.abs() < 0.2);
/// ```
pub fn truncated_normal_fill(
    values: &mut [f64],
    mean: f64,
    std: f64,
    a: f64,
    b: f64,
    rng: &mut InitRng,
) -> Result<()> {
    check_truncated_normal_params(mean, std, a, b)?;
    for value in values.iter_mut() {
        let mut attempts = 0;
        loop {
            let draw = mean + std * rng.next_normal();
            if (a..=b).contains(&draw) {
                *value = draw;
                break;
            }
            attempts += 1;
            if attempts > 1_000_000 {
                return Err(Error::ShapeMismatch {
                    op: "trunc_normal",
                    expected: vec![],
                    got: vec![],
                    msg: alloc::string::String::from(
                        "truncated-normal rejection did not accept a draw; bounds may exclude the mass",
                    ),
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn gram(values: &[f64], rows: usize, cols: usize, transpose_first: bool) -> Vec<Vec<f64>> {
        // Returns Q^T Q (transpose_first) or Q Q^T over the row-major input.
        let (n, m) = if transpose_first {
            (cols, rows)
        } else {
            (rows, cols)
        };
        let mut out = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for k in 0..m {
                    let (a, b) = if transpose_first {
                        (values[k * cols + i], values[k * cols + j])
                    } else {
                        (values[i * cols + k], values[j * cols + k])
                    };
                    dot += a * b;
                }
                out[i][j] = dot;
            }
        }
        out
    }

    fn assert_identity(g: &[Vec<f64>], tol: f64) {
        for (i, row) in g.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (v - expected).abs() < tol,
                    "gram[{i}][{j}] = {v}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn orthogonal_square_is_orthonormal() {
        let mut rng = InitRng::from_seed(1);
        let mut values = vec![0.0; 8 * 8];
        orthogonal_fill(&mut values, 8, 8, 1.0, &mut rng).unwrap();
        assert_identity(&gram(&values, 8, 8, true), 1e-9);
    }

    #[test]
    fn orthogonal_tall_has_orthonormal_columns() {
        let mut rng = InitRng::from_seed(2);
        let mut values = vec![0.0; 9 * 4];
        orthogonal_fill(&mut values, 9, 4, 1.0, &mut rng).unwrap();
        assert_identity(&gram(&values, 9, 4, true), 1e-9);
    }

    #[test]
    fn orthogonal_wide_has_orthonormal_rows() {
        let mut rng = InitRng::from_seed(3);
        let mut values = vec![0.0; 4 * 9];
        orthogonal_fill(&mut values, 4, 9, 1.0, &mut rng).unwrap();
        assert_identity(&gram(&values, 4, 9, false), 1e-9);
    }

    #[test]
    fn orthogonal_gain_scales_gram_by_gain_squared() {
        let mut rng = InitRng::from_seed(4);
        let mut values = vec![0.0; 6 * 5];
        orthogonal_fill(&mut values, 6, 5, 2.0, &mut rng).unwrap();
        let g = gram(&values, 6, 5, true);
        for (i, row) in g.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                let expected = if i == j { 4.0 } else { 0.0 };
                assert!((v - expected).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn orthogonal_is_deterministic_for_seed() {
        let mut a = vec![0.0; 5 * 5];
        let mut b = vec![0.0; 5 * 5];
        orthogonal_fill(&mut a, 5, 5, 1.0, &mut InitRng::from_seed(9)).unwrap();
        orthogonal_fill(&mut b, 5, 5, 1.0, &mut InitRng::from_seed(9)).unwrap();
        assert_eq!(a, b);
        let mut c = vec![0.0; 5 * 5];
        orthogonal_fill(&mut c, 5, 5, 1.0, &mut InitRng::from_seed(10)).unwrap();
        assert_ne!(a, c);
    }

    #[test]
    fn orthogonal_rejects_bad_geometry() {
        let mut rng = InitRng::from_seed(0);
        let mut values = vec![0.0; 4];
        assert!(orthogonal_fill(&mut values, 0, 2, 1.0, &mut rng).is_err());
        assert!(orthogonal_fill(&mut values, 2, 2, f64::NAN, &mut rng).is_err());
        // Slice length must equal rows*cols.
        assert!(orthogonal_fill(&mut values, 3, 3, 1.0, &mut rng).is_err());
    }

    #[test]
    fn trunc_normal_stays_in_bounds_with_approx_moments() {
        let mut rng = InitRng::from_seed(5);
        let n = 20_000;
        let mut values = vec![0.0; n];
        truncated_normal_fill(&mut values, 1.0, 0.5, 0.0, 2.0, &mut rng).unwrap();
        assert!(values.iter().all(|&v| (0.0..=2.0).contains(&v)));
        let mean: f64 = values.iter().sum::<f64>() / n as f64;
        assert!((mean - 1.0).abs() < 0.02, "mean {mean}");
        let var: f64 = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
        // Truncation to [0, 2] = mean +/- 2 std clips the tails: the closed
        // form is 0.25 * (1 - 2*2*phi(2)/(2*Phi(2)-1)) ~= 0.193.
        assert!((var - 0.193).abs() < 0.02, "var {var}");
    }

    #[test]
    fn trunc_normal_rejects_bad_params() {
        let mut rng = InitRng::from_seed(0);
        let mut values = vec![0.0; 4];
        assert!(truncated_normal_fill(&mut values, 0.0, 0.0, -2.0, 2.0, &mut rng).is_err());
        assert!(truncated_normal_fill(&mut values, 0.0, -1.0, -2.0, 2.0, &mut rng).is_err());
        assert!(truncated_normal_fill(&mut values, 0.0, 1.0, 2.0, 2.0, &mut rng).is_err());
        assert!(truncated_normal_fill(&mut values, 0.0, 1.0, 3.0, 2.0, &mut rng).is_err());
        assert!(truncated_normal_fill(&mut values, f64::NAN, 1.0, -2.0, 2.0, &mut rng).is_err());
    }

    #[test]
    fn host_sampled_policies_report_the_execution_gap_at_plan_time() {
        let ctx = InitContext::new(ParameterRole::Weight);
        // Policies validate their parameters first ...
        assert!(orthogonal_with_gain(f64::NAN).plan(ctx).is_err());
        assert!(trunc_normal_with(0.0, 1.0, 2.0, 1.0).plan(ctx).is_err());
        // ... then report the missing host-fill execution arms instead of
        // inventing values.
        let err = orthogonal_().plan(ctx).unwrap_err();
        assert!(alloc::format!("{err:?}").contains("host sampling"));
        let err = trunc_normal_().plan(ctx).unwrap_err();
        assert!(alloc::format!("{err:?}").contains("host sampling"));
    }

    #[test]
    fn context_seed_is_stored_for_the_host_fill_path() {
        let ctx = InitContext::new(ParameterRole::Weight).with_seed(42);
        assert_eq!(ctx.seed, Some(42));
        assert_eq!(InitContext::new(ParameterRole::Weight).seed, None);
        assert_eq!(DEFAULT_HOST_INIT_SEED, 0);
    }

    #[test]
    fn constructors_match_documented_defaults() {
        assert!(matches!(orthogonal_(), Init::Orthogonal { gain } if gain == 1.0));
        assert!(matches!(
            orthogonal_with_gain(0.5),
            Init::Orthogonal { gain } if gain == 0.5
        ));
        assert!(matches!(
            trunc_normal_(),
            Init::TruncatedNormal { mean, std, a, b }
                if mean == 0.0 && std == 1.0 && a == -2.0 && b == 2.0
        ));
        // Existing policies are untouched by the new context field.
        let ctx = InitContext::new(ParameterRole::Weight).with_fan(8, 8);
        assert!(matches!(
            kaiming_uniform().plan(ctx),
            Ok(InitPlan::Uniform { .. })
        ));
    }
}
