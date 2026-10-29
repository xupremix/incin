// Issue #101: `D_MODEL % N_HEADS != 0` is a compile-time failure.
extern crate incin_core as incin;

use incin_backends::cpu::CpuBackendImpl;

fn main() {
    // 10 is not divisible by 4.
    let _ = incin::nn::MultiHeadAttention::<10, 4, 2, CpuBackendImpl>::build(
        incin::nn::AttentionConfig::default(),
        (),
        (),
    );
}
