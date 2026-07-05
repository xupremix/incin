#[derive(Debug, Clone, Copy)]
pub enum Init {
    Zeros,
    Ones,
    Rand,
    Randn,
    KaimingUniform { fan_in: usize, a: f64 },
    KaimingNormal { fan_in: usize, a: f64 },
    XavierUniform { fan_in: usize, fan_out: usize },
    XavierNormal { fan_in: usize, fan_out: usize },
    Constant(f64),
    Uniform { bound: f64 },
}

impl Default for Init {
    fn default() -> Self {
        Self::Zeros
    }
}
