#[derive(Debug, Clone, Copy, Default)]
/// Auto-generated documentation for Init.
pub enum Init {
    #[default]
    /// Auto-generated documentation for Zeros.
    Zeros,
    /// Auto-generated documentation for Ones.
    Ones,
    /// Auto-generated documentation for Rand.
    Rand,
    /// Auto-generated documentation for Randn.
    Randn,
    /// Auto-generated documentation for KaimingUniform.
    KaimingUniform {
        /// fan_in
        fan_in: usize,
        /// a
        a: f64,
    },
    /// Auto-generated documentation for KaimingNormal.
    KaimingNormal {
        /// fan_in
        fan_in: usize,
        /// a
        a: f64,
    },
    /// Auto-generated documentation for XavierUniform.
    XavierUniform {
        /// fan_in
        fan_in: usize,
        /// fan_out
        fan_out: usize,
    },
    /// Auto-generated documentation for XavierNormal.
    XavierNormal {
        /// fan_in
        fan_in: usize,
        /// fan_out
        fan_out: usize,
    },
    /// Auto-generated documentation for Constant.
    Constant(f64),
    /// Auto-generated documentation for Uniform.
    Uniform {
        /// bound
        bound: f64,
    },
}
