#[derive(Debug, Clone, Copy, Default)]
/// `Init`.
pub enum Init {
    #[default]
    /// `Zeros`.
    Zeros,
    /// `Ones`.
    Ones,
    /// `Rand`.
    Rand,
    /// `Randn`.
    Randn,
    /// `KaimingUniform`.
    KaimingUniform {
        /// fan_in
        fan_in: usize,
        /// a
        a: f64,
    },
    /// `KaimingNormal`.
    KaimingNormal {
        /// fan_in
        fan_in: usize,
        /// a
        a: f64,
    },
    /// `XavierUniform`.
    XavierUniform {
        /// fan_in
        fan_in: usize,
        /// fan_out
        fan_out: usize,
    },
    /// `XavierNormal`.
    XavierNormal {
        /// fan_in
        fan_in: usize,
        /// fan_out
        fan_out: usize,
    },
    /// `Constant`.
    Constant(f64),
    /// `Uniform`.
    Uniform {
        /// bound
        bound: f64,
    },
}
