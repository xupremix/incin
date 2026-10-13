use incin::DefaultBackend;
use incin::nn::{AttentionConfig, MultiHeadAttention};

/// A head configuration that satisfies every const assert: 8 wide, 4 query
/// heads, 2 key/value heads.
///
/// This fixture exists so the trybuild project contains a `pass` case:
/// trybuild compiles fail-only sets with `cargo check`, which never
/// monomorphizes `MultiHeadAttention::build` and so never evaluates the
/// `const { assert!(..) }` head invariants. With a `pass` case present it
/// uses `cargo build`, and the mismatch fixtures below fail on the asserts.
fn main() {
    MultiHeadAttention::<8, 4, 2, DefaultBackend>::build(AttentionConfig::default(), (), ())
        .expect("a divisible head configuration must build");
}
