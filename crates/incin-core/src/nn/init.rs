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
}

impl InitContext {
    /// Creates a new `InitContext` with the specified role and no fan info.
    pub const fn new(role: ParameterRole) -> Self {
        Self { role, fan: None }
    }

    /// Attaches fan geometry to this context.
    pub const fn with_fan(mut self, fan_in: usize, fan_out: usize) -> Self {
        self.fan = Some(Fan { fan_in, fan_out });
        self
    }
}

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
