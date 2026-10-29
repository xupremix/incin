use incin::DefaultBackend;
use incin::nn::{AttentionConfig, MultiHeadAttention};

fn main() {
    // 10 is not divisible by 3: D_MODEL % N_HEADS != 0.
    let _ = MultiHeadAttention::<10, 3, 1, DefaultBackend>::build(
        AttentionConfig::default(),
        (),
        (),
    );
}
