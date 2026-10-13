use incin::DefaultBackend;
use incin::nn::{AttentionConfig, MultiHeadAttention};

fn main() {
    // 6 query heads over 4 key/value heads: N_HEADS % N_KV_HEADS != 0.
    let _ = MultiHeadAttention::<12, 6, 4, DefaultBackend>::build(
        AttentionConfig::default(),
        (),
        (),
    );
}
