#[derive(Debug, Clone, Copy, Default)]
/// Core abstraction for `Init` within the Kindle framework..
pub enum Init {
    #[default]
    /// Core abstraction for `Zeros` within the Kindle framework..
    Zeros,
    /// Core abstraction for `Ones` within the Kindle framework..
    Ones,
    /// Core abstraction for `Rand` within the Kindle framework..
    Rand,
    /// Core abstraction for `Randn` within the Kindle framework..
    Randn,
    /// Core abstraction for `KaimingUniform` within the Kindle framework..
    KaimingUniform {
        /// fan_in
        fan_in: usize,
        /// a
        a: f64,
    },
    /// Core abstraction for `KaimingNormal` within the Kindle framework..
    KaimingNormal {
        /// fan_in
        fan_in: usize,
        /// a
        a: f64,
    },
    /// Core abstraction for `XavierUniform` within the Kindle framework..
    XavierUniform {
        /// fan_in
        fan_in: usize,
        /// fan_out
        fan_out: usize,
    },
    /// Core abstraction for `XavierNormal` within the Kindle framework..
    XavierNormal {
        /// fan_in
        fan_in: usize,
        /// fan_out
        fan_out: usize,
    },
    /// Core abstraction for `Constant` within the Kindle framework..
    Constant(f64),
    /// Core abstraction for `Uniform` within the Kindle framework..
    Uniform {
        /// bound
        bound: f64,
    },
}
