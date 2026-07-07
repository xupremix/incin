#[derive(Debug, Clone, Copy, Default)]
pub enum Init {
    #[default]
    Zeros,
    Ones,
    Rand,
    Randn,
    KaimingUniform {
        fan_in: usize,
        a: f64,
    },
    KaimingNormal {
        fan_in: usize,
        a: f64,
    },
    XavierUniform {
        fan_in: usize,
        fan_out: usize,
    },
    XavierNormal {
        fan_in: usize,
        fan_out: usize,
    },
    Constant(f64),
    Uniform {
        bound: f64,
    },
}
