use incin_macros::autotune;

#[autotune(
    key = "valid_key",
    params = [],
    policy = heuristic
)]
fn sample_func() {}

fn main() {}
