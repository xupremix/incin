// Issue #101: `N_HEADS % N_KV_HEADS != 0` is a compile-time failure.
extern crate incin_core as incin;

use incin_backends::cpu::CpuBackendImpl;

fn main() {
    // Four query heads cannot be split into three key/value groups.
    let _ = incin::nn::MultiHeadAttention::<8, 4, 3, CpuBackendImpl>::build(
        incin::nn::AttentionConfig::default(),
        (),
        (),
    );
}
